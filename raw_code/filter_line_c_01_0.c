static void filter_line_c(uint8_t *dst,

                          uint8_t *prev, uint8_t *cur, uint8_t *next,

                          int w, int refs, int parity, int mode)

{

    int x;

    uint8_t *prev2 = parity ? prev : cur ;

    uint8_t *next2 = parity ? cur  : next;

    for (x = 0;  x < w; x++) {

        int c = cur[-refs];

        int d = (prev2[0] + next2[0])>>1;

        int e = cur[+refs];

        int temporal_diff0 = FFABS(prev2[0] - next2[0]);

        int temporal_diff1 =(FFABS(prev[-refs] - c) + FFABS(prev[+refs] - e) )>>1;

        int temporal_diff2 =(FFABS(next[-refs] - c) + FFABS(next[+refs] - e) )>>1;

        int diff = FFMAX3(temporal_diff0>>1, temporal_diff1, temporal_diff2);

        int spatial_pred = (c+e)>>1;

        int spatial_score = FFABS(cur[-refs-1] - cur[+refs-1]) + FFABS(c-e)

                          + FFABS(cur[-refs+1] - cur[+refs+1]) - 1;



#define CHECK(j)\

    {   int score = FFABS(cur[-refs-1+j] - cur[+refs-1-j])\

                  + FFABS(cur[-refs  +j] - cur[+refs  -j])\

                  + FFABS(cur[-refs+1+j] - cur[+refs+1-j]);\

        if (score < spatial_score) {\

            spatial_score= score;\

            spatial_pred= (cur[-refs  +j] + cur[+refs  -j])>>1;\



        CHECK(-1) CHECK(-2) }} }}
