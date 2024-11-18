static inline int mix_core(uint32_t multbl[][256], int a, int b, int c, int d){

#if CONFIG_SMALL

#define ROT(x,s) ((x<<s)|(x>>(32-s)))

    return multbl[0][a] ^ ROT(multbl[0][b], 8) ^ ROT(multbl[0][c], 16) ^ ROT(multbl[0][d], 24);

#else

    return multbl[0][a] ^ multbl[1][b] ^ multbl[2][c] ^ multbl[3][d];

#endif

}
