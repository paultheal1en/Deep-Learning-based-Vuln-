av_cold void ff_huffyuvdsp_init_ppc(HuffYUVDSPContext *c)

{

#if HAVE_ALTIVEC && HAVE_BIGENDIAN

    if (!PPC_ALTIVEC(av_get_cpu_flags()))

        return;



    c->add_bytes = add_bytes_altivec;

#endif /* HAVE_ALTIVEC && HAVE_BIGENDIAN */

}
