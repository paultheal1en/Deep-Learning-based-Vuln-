av_cold void ff_lpc_init_x86(LPCContext *c)

{

#if HAVE_SSE2_INLINE

    int cpu_flags = av_get_cpu_flags();



    if (INLINE_SSE2(cpu_flags) && (cpu_flags & AV_CPU_FLAG_SSE2SLOW)) {

        c->lpc_apply_welch_window = lpc_apply_welch_window_sse2;

        c->lpc_compute_autocorr   = lpc_compute_autocorr_sse2;

    }

#endif /* HAVE_SSE2_INLINE */

}
