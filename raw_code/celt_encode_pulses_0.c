static inline void celt_encode_pulses(OpusRangeCoder *rc, int *y, uint32_t N, uint32_t K)

{

    ff_opus_rc_enc_uint(rc, celt_icwrsi(N, y), CELT_PVQ_V(N, K));

}
