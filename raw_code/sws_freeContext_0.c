void sws_freeContext(SwsContext *c)

{

    int i;

    if (!c) return;



    if (c->lumPixBuf) {

        for (i=0; i<c->vLumBufSize; i++)

            av_freep(&c->lumPixBuf[i]);

        av_freep(&c->lumPixBuf);

    }



    if (c->chrPixBuf) {

        for (i=0; i<c->vChrBufSize; i++)

            av_freep(&c->chrPixBuf[i]);

        av_freep(&c->chrPixBuf);

    }



    if (CONFIG_SWSCALE_ALPHA && c->alpPixBuf) {

        for (i=0; i<c->vLumBufSize; i++)

            av_freep(&c->alpPixBuf[i]);

        av_freep(&c->alpPixBuf);

    }



    av_freep(&c->vLumFilter);

    av_freep(&c->vChrFilter);

    av_freep(&c->hLumFilter);

    av_freep(&c->hChrFilter);

#if ARCH_PPC && HAVE_ALTIVEC

    av_freep(&c->vYCoeffsBank);

    av_freep(&c->vCCoeffsBank);

#endif



    av_freep(&c->vLumFilterPos);

    av_freep(&c->vChrFilterPos);

    av_freep(&c->hLumFilterPos);

    av_freep(&c->hChrFilterPos);



#if ARCH_X86 && CONFIG_GPL

#ifdef MAP_ANONYMOUS

    if (c->lumMmx2FilterCode) munmap(c->lumMmx2FilterCode, c->lumMmx2FilterCodeSize);

    if (c->chrMmx2FilterCode) munmap(c->chrMmx2FilterCode, c->chrMmx2FilterCodeSize);

#elif HAVE_VIRTUALALLOC

    if (c->lumMmx2FilterCode) VirtualFree(c->lumMmx2FilterCode, 0, MEM_RELEASE);

    if (c->chrMmx2FilterCode) VirtualFree(c->chrMmx2FilterCode, 0, MEM_RELEASE);

#else

    av_free(c->lumMmx2FilterCode);

    av_free(c->chrMmx2FilterCode);

#endif

    c->lumMmx2FilterCode=NULL;

    c->chrMmx2FilterCode=NULL;

#endif /* ARCH_X86 && CONFIG_GPL */



    av_freep(&c->yuvTable);



    av_free(c);

}
