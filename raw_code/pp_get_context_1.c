pp_context *pp_get_context(int width, int height, int cpuCaps){
    PPContext *c= av_mallocz(sizeof(PPContext));
    int stride= FFALIGN(width, 16);  //assumed / will realloc if needed
    int qpStride= (width+15)/16 + 2; //assumed / will realloc if needed
    c->av_class = &av_codec_context_class;
    if(cpuCaps&PP_FORMAT){
        c->hChromaSubSample= cpuCaps&0x3;
        c->vChromaSubSample= (cpuCaps>>4)&0x3;
    }else{
        c->hChromaSubSample= 1;
        c->vChromaSubSample= 1;
    }
    if (cpuCaps & PP_CPU_CAPS_AUTO) {
        c->cpuCaps = av_get_cpu_flags();
    } else {
        c->cpuCaps = 0;
        if (cpuCaps & PP_CPU_CAPS_MMX)      c->cpuCaps |= AV_CPU_FLAG_MMX;
        if (cpuCaps & PP_CPU_CAPS_MMX2)     c->cpuCaps |= AV_CPU_FLAG_MMXEXT;
        if (cpuCaps & PP_CPU_CAPS_3DNOW)    c->cpuCaps |= AV_CPU_FLAG_3DNOW;
        if (cpuCaps & PP_CPU_CAPS_ALTIVEC)  c->cpuCaps |= AV_CPU_FLAG_ALTIVEC;
    }
    reallocBuffers(c, width, height, stride, qpStride);
    c->frameNum=-1;
    return c;
}