// ANS-to-LDS decode benchmark.
//
// Decodes a native hipANS archive with the library's own decoder
// (ANSDecodeWarpFullBlock) and keeps the decoded bytes in LDS, which is what a
// fused decode+GEMM kernel would consume. Optimizations are selected at build
// time (HIPANS_DECODE_WARP_BARRIER, HIPANS_BLOCK_SIZE) and at run time
// (--input lds stages the compressed block in LDS first). See
// docs/OPTIMIZATIONS.md.
#include "hipans/ans/GpuANSDecode.h"
#include "hipans/ans/GpuANSStageLds.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef HIPANS_COMMIT_BUILT
#define HIPANS_COMMIT_BUILT "unknown"
#endif
#ifndef HIPANS_BUILD_FLAGS
#define HIPANS_BUILD_FLAGS "unknown"
#endif
namespace ha = hipans;
static_assert(ha::kDefaultBlockSize >= 128 && ha::kDefaultBlockSize % 128 == 0,
              "hipANS blocks must contain complete 128-byte rows");
static std::string jsonEscape(const std::string& value) {
  static const char hex[] = "0123456789abcdef";
  std::string out;
  for (unsigned char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out += hex[c >> 4];
          out += hex[c & 0x0f];
        } else out += static_cast<char>(c);
    }
  }
  return out;
}
static void check(hipError_t e) { if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) { check(hipMalloc(reinterpret_cast<void**>(&p),std::max(size_t(1),n)*sizeof(T))); }
  ~Device() { if(p)(void)hipFree(p); }
};
static std::vector<uint8_t> read(const std::string& path) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f)throw std::runtime_error("cannot open "+path);
  auto n=f.tellg();if(n<0)throw std::runtime_error("invalid file length");
  std::vector<uint8_t> b(static_cast<size_t>(n));f.seekg(0);
  f.read(reinterpret_cast<char*>(b.data()),b.size());if(!f)throw std::runtime_error("short file");return b;
}
// Same matrix-to-tile mapping as the real-weight extraction delivery, not codec logic.
static std::vector<uint8_t> tiles(const std::vector<uint8_t>& raw,uint32_t rows,uint32_t cols) {
  if(raw.size()!=uint64_t(rows)*cols)throw std::runtime_error("reference shape mismatch");
  constexpr uint32_t blockRows=ha::kDefaultBlockSize/128;
  if(rows%blockRows||cols%128)throw std::runtime_error("matrix shape must align to full hipANS tiles");
  uint64_t nr=rows/blockRows,nc=cols/128;
  if(nr*nc*ha::kDefaultBlockSize>uint64_t(INT32_MAX))throw std::runtime_error("archive exceeds hipANS single-input limit");
  std::vector<uint8_t> out(nr*nc*ha::kDefaultBlockSize,0);
  for(uint64_t t=0;t<nr*nc;t++)for(uint32_t i=0;i<ha::kDefaultBlockSize;i++) {
    uint64_t r=t/nc*blockRows+i/128,c=t%nc*128+i%128;
    out[t*ha::kDefaultBlockSize+i]=raw[r*cols+c];
  }
  return out;
}
struct LDSWriter {
  volatile uint8_t* tile;
  bool active;
  __device__ void write(uint32_t offset,uint8_t sym) {
    if(active) {
      uint32_t address=uint32_t(reinterpret_cast<uintptr_t>(tile+offset));
      uint32_t value=sym;
      asm volatile("ds_write_b8 %0, %1" :: "v"(address),"v"(value) : "memory");
    }
  }
};

template<int Threads,bool Verify,bool Raw,bool StageLds>
__global__ __launch_bounds__(Threads) void hipansLDS(
    const uint8_t* archives,uint64_t stride,const ha::TableT* table,
    const uint8_t* raw,uint8_t* output,uint32_t blocks,uint32_t replicas,
    uint64_t groupBegin,uint64_t groupEnd,uint32_t stageWordsStride) {
  extern __shared__ __align__(16) uint8_t storage[];
  auto lookup=reinterpret_cast<ha::TableT*>(storage);
  volatile uint8_t* tile=storage+(Raw?0:4096);
  unsigned tid=threadIdx.x,warp=tid/64,lane=ha::getLaneId();
  constexpr unsigned Waves=Threads/64;
  if constexpr(!Raw) {
    // Preserve the original hipANS uint4 LUT staging; only loop bounds support
    // additional deployment sizes. Probability precision remains the default 10.
    auto dst=reinterpret_cast<uint4*>(lookup);
    auto src=reinterpret_cast<const uint4*>(table);
    for(unsigned j=tid;j<256;j+=Threads)dst[j]=src[j];
    __syncthreads();
  }
  const uint64_t total=uint64_t(blocks)*replicas;
  for(uint64_t group=groupBegin+blockIdx.x;group<groupEnd;group+=gridDim.x) {
    uint64_t task=group*Waves+warp;
    bool active=task<total;
    // All waves must execute equal full-block paths because the decoder may
    // have workgroup barriers inside decodeOneWarp. Tail waves decode block 0,
    // discard it, and never contribute to the effective numerator.
    uint64_t safeTask=active?task:0;
    unsigned rep=safeTask/blocks,block=safeTask%blocks;
    LDSWriter writer{tile+warp*ha::kDefaultBlockSize,active};
    if constexpr(Raw) {
      for(unsigned i=lane;i<ha::kDefaultBlockSize;i+=64)writer.write(i,raw[safeTask*ha::kDefaultBlockSize+i]);
    } else {
      auto header=reinterpret_cast<const ha::ANSCoalescedHeader*>(archives+uint64_t(rep)*stride);
      auto state=header->getWarpStates()[block].warpState[lane];
      auto words=header->getBlockWords(blocks)[block];
      auto data=header->getBlockDataStart(blocks)+words.y;
      if constexpr(StageLds) {
        auto staging=reinterpret_cast<ha::ANSEncodedT*>(
            storage+4096+Waves*ha::kDefaultBlockSize+
            size_t(warp)*stageWordsStride*sizeof(ha::ANSEncodedT));
        ha::stageCompressedBlockToLds(data,staging,words.x&0xffffU,lane);
        // Publish the per-wave compressed block exactly once before decoding.
        __syncthreads();
        data=staging;
      }
      // The library's own decoder: state recurrence, 16-bit refill and
      // ballot/prefix; the per-symbol barrier is present unless
      // HIPANS_DECODE_WARP_BARRIER=0.
      ha::ANSDecodeWarpFullBlock<LDSWriter,10,ha::kDefaultBlockSize,false>::decode(
          lane,state,words.x&0xffff,data,writer,lookup,block);
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __syncthreads();
    if constexpr(Verify) {
      for(unsigned i=tid;i<Waves*ha::kDefaultBlockSize;i+=Threads) {
        uint64_t owner=group*Waves+i/ha::kDefaultBlockSize;
        if(owner<total)output[owner*ha::kDefaultBlockSize+i%ha::kDefaultBlockSize]=tile[i];
      }
      __syncthreads();
    }
  }
}
template<int T> static const void* kernel(bool verify,bool raw,bool stageLds) {
  if(stageLds) {
    if(raw)throw std::runtime_error("--input lds is only valid for compressed decode mode");
    return verify?reinterpret_cast<const void*>(hipansLDS<T,true,false,true>):reinterpret_cast<const void*>(hipansLDS<T,false,false,true>);
  }
  if(raw)return verify?reinterpret_cast<const void*>(hipansLDS<T,true,true,false>):reinterpret_cast<const void*>(hipansLDS<T,false,true,false>);
  return verify?reinterpret_cast<const void*>(hipansLDS<T,true,false,false>):reinterpret_cast<const void*>(hipansLDS<T,false,false,false>);
}
static const void* select(int t,bool verify,bool raw,bool stageLds) {
  switch(t) {
#define CASE(T) case T:return kernel<T>(verify,raw,stageLds)
    CASE(64);CASE(128);CASE(192);CASE(256);CASE(320);CASE(384);CASE(448);
#undef CASE
  }
  throw std::runtime_error("threads must be 64,128,192,256,320,384,448");
}
struct Options {
  std::string artifact,reference,mode="decode",input="global";
  uint32_t rows=0,cols=0,replicas=1,threads=128,gridMult=4,streams=1,warmup=5,iters=30;
  bool prepare=false,replace=false,skip=false;
};
static Options options(int argc,char**argv) {
  Options a;
  for(int i=1;i<argc;i++) {
    std::string k=argv[i];
    if(k=="--prepare"){a.prepare=true;continue;}
    if(k=="--replace"){a.replace=true;continue;}
    if(k=="--skip-verify"){a.skip=true;continue;}
    if(i+1>=argc)throw std::runtime_error("missing argument value");std::string v=argv[++i];
    if(k=="--artifact")a.artifact=v;else if(k=="--reference")a.reference=v;
    else if(k=="--mode")a.mode=v;
    else if(k=="--input")a.input=v;
    else {
      size_t end;uint64_t n=std::stoull(v,&end);
      if(end!=v.size()||n>UINT32_MAX||v[0]=='-')throw std::runtime_error("invalid numeric argument");
      if(k=="--rows")a.rows=n;else if(k=="--cols")a.cols=n;
      else if(k=="--replicas")a.replicas=n;else if(k=="--threads")a.threads=n;
      else if(k=="--grid-mult")a.gridMult=n;else if(k=="--streams")a.streams=n;
      else if(k=="--warmup")a.warmup=n;else if(k=="--iters")a.iters=n;
      else throw std::runtime_error("unknown option "+k);
    }
  }
  if(a.artifact.empty()||a.reference.empty()||!a.rows||!a.cols||!a.replicas||a.replicas>256||
      !a.gridMult||a.gridMult>16||!a.streams||a.streams>32||!a.iters||
      (a.mode!="decode"&&a.mode!="raw")||(a.input!="global"&&a.input!="lds")||
      (a.input=="lds"&&a.mode!="decode")||
      (a.prepare&&(a.skip||a.mode!="decode"||a.input!="global"||a.replicas!=1)))
    throw std::runtime_error("invalid CLI configuration");
  select(a.threads,false,a.mode=="raw",a.input=="lds");return a;
}
static ha::ANSCoalescedHeader validate(const std::vector<uint8_t>& b,size_t expected) {
  if(b.size()<32)throw std::runtime_error("short native archive");
  ha::ANSCoalescedHeader h;std::memcpy(&h,b.data(),32);
  uint32_t magic;std::memcpy(&magic,b.data(),4);
  if(expected%ha::kDefaultBlockSize||magic!=((ha::kANSMagic<<16)|ha::kANSVersion)||h.getProbBits()!=10||
      h.getUseChecksum()||h.getTotalUncompressedWords()!=expected||
      h.getNumBlocks()!=expected/ha::kDefaultBlockSize||h.getTotalCompressedSize()!=b.size())
    throw std::runtime_error("archive is not the expected native hipANS p10 full-tile artifact");
  uint16_t probs[256];std::memcpy(probs,b.data()+32,512);uint32_t sum=0;
  for(auto p:probs)sum+=p;if(sum!=1024)throw std::runtime_error("invalid probability sum");
  size_t descriptors=32+512+size_t(h.getNumBlocks())*sizeof(ha::ANSWarpState);
  for(uint32_t i=0;i<h.getNumBlocks();i++) {
    uint2 w;std::memcpy(&w,b.data()+descriptors+i*8,8);
    if((w.x>>16)!=ha::kDefaultBlockSize||uint64_t(w.y)+(w.x&0xffff)>h.getTotalCompressedWords())
      throw std::runtime_error("invalid native block descriptor");
  }
  return h;
}
static uint32_t findMaxCompressedWords(const std::vector<uint8_t>& archive,uint32_t blocks) {
  const size_t offset=sizeof(ha::ANSCoalescedHeader)+ha::kNumSymbols*sizeof(uint16_t)+
      size_t(blocks)*sizeof(ha::ANSWarpState);
  uint32_t maxWords=0;
  for(uint32_t b=0;b<blocks;b++) {
    uint2 descriptor;
    std::memcpy(&descriptor,archive.data()+offset+size_t(b)*sizeof(uint2),sizeof(descriptor));
    maxWords=std::max(maxWords,uint32_t(descriptor.x&0xffffU));
  }
  return maxWords;
}
static bool compressedBlockStartsAligned(const std::vector<uint8_t>& archive,uint32_t blocks) {
  const size_t offset=sizeof(ha::ANSCoalescedHeader)+ha::kNumSymbols*sizeof(uint16_t)+
      size_t(blocks)*sizeof(ha::ANSWarpState);
  const uint64_t payloadOffset=ha::ANSCoalescedHeader::getCompressedOverhead(blocks);
  for(uint32_t b=0;b<blocks;b++) {
    uint2 descriptor;
    std::memcpy(&descriptor,archive.data()+offset+size_t(b)*sizeof(uint2),sizeof(descriptor));
    const uint64_t byteOffset=payloadOffset+uint64_t(descriptor.y)*sizeof(ha::ANSEncodedT);
    if(byteOffset%16!=0)return false;
  }
  return true;
}
int main(int argc,char**argv) {try {
  auto a=options(argc,argv);auto raw=read(a.reference);auto tiled=tiles(raw,a.rows,a.cols);
  int dev;check(hipGetDevice(&dev));hipDeviceProp_t props;check(hipGetDeviceProperties(&props,dev));
  if(props.warpSize!=64)throw std::runtime_error("GPU must be a wave64 device");
  hipStream_t control;check(hipStreamCreateWithFlags(&control,hipStreamNonBlocking));
  std::vector<uint8_t> archive;
  if(a.prepare) {
    if(std::ifstream(a.artifact)&&!a.replace)throw std::runtime_error("artifact exists; explicit --replace required");
    Device<uint8_t> input(tiled.size()),encoded(ha::getMaxCompressedSize(tiled.size()));Device<uint32_t> size(1);
    check(hipMemcpyAsync(input.p,tiled.data(),tiled.size(),hipMemcpyHostToDevice,control));
    auto stack=ha::makeStackMemory(1ull<<30);
    const void* in=input.p;void* out=encoded.p;uint32_t bytes=tiled.size();
    ha::ansEncodeBatchPointer(stack,ha::ANSCodecConfig(10,false),1,&in,&bytes,nullptr,&out,size.p,control);
    check(hipStreamSynchronize(control));uint32_t n;
    check(hipMemcpy(&n,size.p,4,hipMemcpyDeviceToHost));
    if(n>ha::getMaxCompressedSize(tiled.size()))throw std::runtime_error("encoded size exceeds allocation");
    archive.resize(n);check(hipMemcpy(archive.data(),encoded.p,n,hipMemcpyDeviceToHost));
  } else archive=read(a.artifact);
  auto h=validate(archive,tiled.size());uint32_t blocks=h.getNumBlocks();
  const uint32_t maxCompressedWords=a.input=="lds"?findMaxCompressedWords(archive,blocks):0;
  const uint32_t stageWordsStride=(maxCompressedWords+7U)&~7U;
  if(a.input=="lds"&&!compressedBlockStartsAligned(archive,blocks))
    throw std::runtime_error("--input lds requires 16-byte-aligned compressed block starts");
  uint64_t stride=(archive.size()+31)&~uint64_t(31),totalBytes=uint64_t(tiled.size())*a.replicas;
  Device<uint8_t> archives(stride*a.replicas),rawDevice(a.mode=="raw"?totalBytes:0),output(a.skip?0:totalBytes);
  Device<ha::TableT> lookup(1024);
  for(uint32_t r=0;r<a.replicas;r++) {
    check(hipMemcpyAsync(archives.p+r*stride,archive.data(),archive.size(),hipMemcpyHostToDevice,control));
    if(a.mode=="raw")check(hipMemcpyAsync(rawDevice.p+uint64_t(r)*tiled.size(),tiled.data(),tiled.size(),hipMemcpyHostToDevice,control));
  }
  ha::BatchProviderStride provider(archives.p,stride);
  hipEvent_t begin,end;check(hipEventCreate(&begin));check(hipEventCreate(&end));
  check(hipEventRecord(begin,control));
  hipLaunchKernelGGL((ha::ansDecodeTable<ha::BatchProviderStride,512>),dim3(1),dim3(512),0,control,provider,10,lookup.p);
  check(hipGetLastError());check(hipEventRecord(end,control));check(hipEventSynchronize(end));
  float tableMs;check(hipEventElapsedTime(&tableMs,begin,end));
  bool rawMode=a.mode=="raw",stageMode=a.input=="lds";
  size_t stageShared=stageMode?size_t(a.threads/64)*stageWordsStride*sizeof(ha::ANSEncodedT):0;
  size_t shared=(a.threads/64)*ha::kDefaultBlockSize+(rawMode?0:4096)+stageShared;
  if(shared>props.sharedMemPerBlock)throw std::runtime_error("LDS allocation exceeds device limit");
  const void* timedKernel=select(a.threads,false,rawMode,stageMode);int activeBlocks;
  check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks,timedKernel,a.threads,shared));
  if(!activeBlocks)throw std::runtime_error("no resident workgroups");
  hipFuncAttributes attrs;check(hipFuncGetAttributes(&attrs,timedKernel));
  uint64_t groups=(uint64_t(blocks)*a.replicas+(a.threads/64)-1)/(a.threads/64);
  uint64_t workgroupSlots=uint64_t(activeBlocks)*props.multiProcessorCount;
  std::vector<hipStream_t> streams(a.streams);std::vector<hipEvent_t> finished(a.streams);
  for(unsigned s=0;s<a.streams;s++){check(hipStreamCreateWithFlags(&streams[s],hipStreamNonBlocking));check(hipEventCreateWithFlags(&finished[s],hipEventDisableTiming));}
  unsigned gridCap=activeBlocks*props.multiProcessorCount*a.gridMult;
  std::vector<unsigned> grids(a.streams);
  auto launch=[&](bool verify) {
    const void* f=select(a.threads,verify,rawMode,stageMode);
    for(unsigned s=0;s<a.streams;s++) {
      uint64_t first=groups*s/a.streams,last=groups*(s+1)/a.streams;
      grids[s]=std::min(uint64_t(gridCap),last-first);
      if(!grids[s])continue;
      auto ap=archives.p;auto tp=lookup.p;auto rp=rawDevice.p;auto op=output.p;
      void* args[]={&ap,&stride,&tp,&rp,&op,&blocks,&a.replicas,&first,&last,
                    const_cast<uint32_t*>(&stageWordsStride)};
      check(hipLaunchKernel(f,dim3(grids[s]),dim3(a.threads),args,shared,streams[s]));
    }
  };
  auto synchronize=[&](){for(auto stream:streams)check(hipStreamSynchronize(stream));};
  if(!a.skip) {
    launch(true);synchronize();std::vector<uint8_t> recovered(tiled.size());
    for(unsigned r=0;r<a.replicas;r++) {
      check(hipMemcpy(recovered.data(),output.p+uint64_t(r)*tiled.size(),recovered.size(),hipMemcpyDeviceToHost));
      if(recovered!=tiled)throw std::runtime_error("actual LDS byte-exact hipANS roundtrip failed");
    }
  }
  for(unsigned i=0;i<a.warmup;i++)launch(false);synchronize();
  std::vector<double> samples;
  for(unsigned i=0;i<a.iters;i++) {
    check(hipEventRecord(begin,control));
    for(auto stream:streams)check(hipStreamWaitEvent(stream,begin,0));
    launch(false);
    for(unsigned s=0;s<a.streams;s++){check(hipEventRecord(finished[s],streams[s]));check(hipStreamWaitEvent(control,finished[s],0));}
    check(hipEventRecord(end,control));check(hipEventSynchronize(end));float ms;
    check(hipEventElapsedTime(&ms,begin,end));samples.push_back(ms);
  }
  std::sort(samples.begin(),samples.end());auto pct=[&](double p){double x=p*(samples.size()-1);size_t lo=x;return samples[lo]+(samples[std::min(lo+1,samples.size()-1)]-samples[lo])*(x-lo);};
  double median=pct(.5);if(median<=0)throw std::runtime_error("invalid latency");
  if(a.prepare) {
    std::string temporary=a.artifact+".pending";std::ofstream out(temporary,std::ios::binary);
    out.write(reinterpret_cast<char*>(archive.data()),archive.size());out.close();
    if(!out||std::rename(temporary.c_str(),a.artifact.c_str()))throw std::runtime_error("artifact save failed");
  }
  uint64_t effective=uint64_t(raw.size())*a.replicas;
  std::cout<<std::setprecision(10)<<"{\"schema\":\"hipans-lds-v1\",\"codec\":\"hipANS ANSDecodeWarpFullBlock + decodeOneWarp\",\"mode\":\""<<a.mode
    <<"\",\"input\":\""<<a.input<<"\",\"verified_actual_LDS\":"<<(a.skip?"false":"true")<<",\"prepare\":"<<(a.prepare?"true":"false")
    <<",\"unique_weight_bytes\":"<<raw.size()<<",\"effective_decoded_bytes\":"<<effective<<",\"padded_decoded_bytes\":"<<totalBytes
    <<",\"replicas\":"<<a.replicas<<",\"block_size_bytes\":"<<ha::kDefaultBlockSize<<",\"block_rows\":"<<ha::kDefaultBlockSize/128
    <<",\"num_blocks\":"<<blocks<<",\"artifact_bytes\":"<<archive.size()<<",\"artifact_metadata_bytes\":"<<h.getCompressedOverhead()
    <<",\"compressed_allocation_bytes\":"<<stride*a.replicas<<",\"ratio_codes\":"<<double(raw.size())/archive.size()
    <<",\"compression_ratio_including_metadata\":"<<double(raw.size())/archive.size()
    <<",\"decode_ms\":{\"median\":"<<median<<",\"p10\":"<<pct(.1)<<",\"p90\":"<<pct(.9)<<"}"
    <<",\"effective_LDS_GBps\":"<<effective/median/1e6<<",\"padded_LDS_GBps\":"<<totalBytes/median/1e6
    <<",\"compressed_archive_GBps\":"<<(rawMode?0:archive.size()*a.replicas/median/1e6)
    <<",\"lookup_prepare_ms_outside_timing\":"<<tableMs<<",\"warmup\":"<<a.warmup<<",\"iters\":"<<a.iters
    <<",\"timing\":\"common start; all queues joined before stop; not sum of queue rates\""
    <<",\"replication_policy\":\"independent device addresses; same real matrix; not additional model parameters\""
    <<",\"launch\":{\"threads\":"<<a.threads<<",\"waves_per_workgroup\":"<<a.threads/64<<",\"grid_multiplier\":"<<a.gridMult<<",\"streams\":"<<a.streams<<",\"grid_per_stream\":[";
  for(unsigned i=0;i<grids.size();i++)std::cout<<(i?",":"")<<grids[i];
  std::cout<<"],\"LDS_bytes\":"<<shared<<",\"stage_words_stride_per_wave\":"<<stageWordsStride
    <<",\"max_compressed_words_per_block\":"<<maxCompressedWords<<",\"stage_LDS_bytes\":"<<stageShared
    <<",\"max_resident_workgroups_per_CU\":"<<activeBlocks
    <<",\"theoretical_resident_waves\":"<<uint64_t(activeBlocks)*props.multiProcessorCount*(a.threads/64)
    <<",\"workgroup_tasks\":"<<groups<<",\"workgroup_slots\":"<<workgroupSlots<<",\"workgroup_rounds\":"<<double(groups)/workgroupSlots
    <<",\"tail_dummy_tiles\":"<<groups*(a.threads/64)-uint64_t(blocks)*a.replicas<<",\"VGPR\":"<<attrs.numRegs<<",\"scratch_bytes\":"<<attrs.localSizeBytes<<"}"
    <<",\"HIPANS_COMMIT\":\""<<jsonEscape(HIPANS_COMMIT_BUILT)
    <<"\",\"provenance\":{\"gpu_name\":\""<<props.name<<"\",\"gpu_arch\":\""<<props.gcnArchName
    <<"\",\"hipANS_commit_built\":\""<<jsonEscape(HIPANS_COMMIT_BUILT)
    <<"\",\"HIPANS_DECODE_WARP_BARRIER\":"<<HIPANS_DECODE_WARP_BARRIER
    <<",\"build_flags\":\""<<jsonEscape(HIPANS_BUILD_FLAGS)<<"\"}}\n";
  for(unsigned s=0;s<a.streams;s++){check(hipEventDestroy(finished[s]));check(hipStreamDestroy(streams[s]));}
  check(hipEventDestroy(begin));check(hipEventDestroy(end));check(hipStreamDestroy(control));
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
