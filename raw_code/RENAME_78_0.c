static void RENAME(yadif_filter_line)(uint8_t *dst, uint8_t *prev, uint8_t *cur,

                                      uint8_t *next, int w, int prefs,

                                      int mrefs, int parity, int mode)

{

    DECLARE_ALIGNED(16, uint8_t, tmp0)[16];

    DECLARE_ALIGNED(16, uint8_t, tmp1)[16];

    DECLARE_ALIGNED(16, uint8_t, tmp2)[16];

    DECLARE_ALIGNED(16, uint8_t, tmp3)[16];

    int x;



#define FILTER\

    for(x=0; x<w; x+=STEP){\

        __asm__ volatile(\

            "pxor      "MM"7, "MM"7 \n\t"\

            LOAD("(%[cur],%[mrefs])", MM"0") /* c = cur[x-refs] */\

            LOAD("(%[cur],%[prefs])", MM"1") /* e = cur[x+refs] */\

            LOAD("(%["prev2"])", MM"2") /* prev2[x] */\

            LOAD("(%["next2"])", MM"3") /* next2[x] */\

            MOVQ"      "MM"3, "MM"4 \n\t"\

            "paddw     "MM"2, "MM"3 \n\t"\

            "psraw     $1,    "MM"3 \n\t" /* d = (prev2[x] + next2[x])>>1 */\

            MOVQ"      "MM"0, %[tmp0] \n\t" /* c */\

            MOVQ"      "MM"3, %[tmp1] \n\t" /* d */\

            MOVQ"      "MM"1, %[tmp2] \n\t" /* e */\

            "psubw     "MM"4, "MM"2 \n\t"\

            PABS(      MM"4", MM"2") /* temporal_diff0 */\

            LOAD("(%[prev],%[mrefs])", MM"3") /* prev[x-refs] */\

            LOAD("(%[prev],%[prefs])", MM"4") /* prev[x+refs] */\

            "psubw     "MM"0, "MM"3 \n\t"\

            "psubw     "MM"1, "MM"4 \n\t"\

            PABS(      MM"5", MM"3")\

            PABS(      MM"5", MM"4")\

            "paddw     "MM"4, "MM"3 \n\t" /* temporal_diff1 */\

            "psrlw     $1,    "MM"2 \n\t"\

            "psrlw     $1,    "MM"3 \n\t"\

            "pmaxsw    "MM"3, "MM"2 \n\t"\

            LOAD("(%[next],%[mrefs])", MM"3") /* next[x-refs] */\

            LOAD("(%[next],%[prefs])", MM"4") /* next[x+refs] */\

            "psubw     "MM"0, "MM"3 \n\t"\

            "psubw     "MM"1, "MM"4 \n\t"\

            PABS(      MM"5", MM"3")\

            PABS(      MM"5", MM"4")\

            "paddw     "MM"4, "MM"3 \n\t" /* temporal_diff2 */\

            "psrlw     $1,    "MM"3 \n\t"\

            "pmaxsw    "MM"3, "MM"2 \n\t"\

            MOVQ"      "MM"2, %[tmp3] \n\t" /* diff */\

\

            "paddw     "MM"0, "MM"1 \n\t"\

            "paddw     "MM"0, "MM"0 \n\t"\

            "psubw     "MM"1, "MM"0 \n\t"\

            "psrlw     $1,    "MM"1 \n\t" /* spatial_pred */\

            PABS(      MM"2", MM"0")      /* ABS(c-e) */\

\

            MOVQU" -1(%[cur],%[mrefs]), "MM"2 \n\t" /* cur[x-refs-1] */\

            MOVQU" -1(%[cur],%[prefs]), "MM"3 \n\t" /* cur[x+refs-1] */\

            MOVQ"      "MM"2, "MM"4 \n\t"\

            "psubusb   "MM"3, "MM"2 \n\t"\

            "psubusb   "MM"4, "MM"3 \n\t"\

            "pmaxub    "MM"3, "MM"2 \n\t"\

            PSHUF(MM"3", MM"2") \

            "punpcklbw "MM"7, "MM"2 \n\t" /* ABS(cur[x-refs-1] - cur[x+refs-1]) */\

            "punpcklbw "MM"7, "MM"3 \n\t" /* ABS(cur[x-refs+1] - cur[x+refs+1]) */\

            "paddw     "MM"2, "MM"0 \n\t"\

            "paddw     "MM"3, "MM"0 \n\t"\

            "psubw    "MANGLE(pw_1)", "MM"0 \n\t" /* spatial_score */\

\

            CHECK(-2,0)\

            CHECK1\

            CHECK(-3,1)\

            CHECK2\

            CHECK(0,-2)\

            CHECK1\

            CHECK(1,-3)\

            CHECK2\

\

            /* if(p->mode<2) ... */\

            MOVQ"    %[tmp3], "MM"6 \n\t" /* diff */\

            "cmpl      $2, %[mode] \n\t"\

            "jge       1f \n\t"\

            LOAD("(%["prev2"],%[mrefs],2)", MM"2") /* prev2[x-2*refs] */\

            LOAD("(%["next2"],%[mrefs],2)", MM"4") /* next2[x-2*refs] */\

            LOAD("(%["prev2"],%[prefs],2)", MM"3") /* prev2[x+2*refs] */\

            LOAD("(%["next2"],%[prefs],2)", MM"5") /* next2[x+2*refs] */\

            "paddw     "MM"4, "MM"2 \n\t"\

            "paddw     "MM"5, "MM"3 \n\t"\

            "psrlw     $1,    "MM"2 \n\t" /* b */\

            "psrlw     $1,    "MM"3 \n\t" /* f */\

            MOVQ"    %[tmp0], "MM"4 \n\t" /* c */\

            MOVQ"    %[tmp1], "MM"5 \n\t" /* d */\

            MOVQ"    %[tmp2], "MM"7 \n\t" /* e */\

            "psubw     "MM"4, "MM"2 \n\t" /* b-c */\

            "psubw     "MM"7, "MM"3 \n\t" /* f-e */\

            MOVQ"      "MM"5, "MM"0 \n\t"\

            "psubw     "MM"4, "MM"5 \n\t" /* d-c */\

            "psubw     "MM"7, "MM"0 \n\t" /* d-e */\

            MOVQ"      "MM"2, "MM"4 \n\t"\

            "pminsw    "MM"3, "MM"2 \n\t"\

            "pmaxsw    "MM"4, "MM"3 \n\t"\

            "pmaxsw    "MM"5, "MM"2 \n\t"\

            "pminsw    "MM"5, "MM"3 \n\t"\

            "pmaxsw    "MM"0, "MM"2 \n\t" /* max */\

            "pminsw    "MM"0, "MM"3 \n\t" /* min */\

            "pxor      "MM"4, "MM"4 \n\t"\

            "pmaxsw    "MM"3, "MM"6 \n\t"\

            "psubw     "MM"2, "MM"4 \n\t" /* -max */\

            "pmaxsw    "MM"4, "MM"6 \n\t" /* diff= MAX3(diff, min, -max); */\

            "1: \n\t"\

\

            MOVQ"    %[tmp1], "MM"2 \n\t" /* d */\

            MOVQ"      "MM"2, "MM"3 \n\t"\

            "psubw     "MM"6, "MM"2 \n\t" /* d-diff */\

            "paddw     "MM"6, "MM"3 \n\t" /* d+diff */\

            "pmaxsw    "MM"2, "MM"1 \n\t"\

            "pminsw    "MM"3, "MM"1 \n\t" /* d = clip(spatial_pred, d-diff, d+diff); */\

            "packuswb  "MM"1, "MM"1 \n\t"\

\

            :[tmp0]"=m"(tmp0),\

             [tmp1]"=m"(tmp1),\

             [tmp2]"=m"(tmp2),\

             [tmp3]"=m"(tmp3)\

            :[prev] "r"(prev),\

             [cur]  "r"(cur),\

             [next] "r"(next),\

             [prefs]"r"((x86_reg)prefs),\

             [mrefs]"r"((x86_reg)mrefs),\

             [mode] "g"(mode)\

        );\

        __asm__ volatile(MOV" "MM"1, %0" :"=m"(*dst));\

        dst += STEP;\

        prev+= STEP;\

        cur += STEP;\

        next+= STEP;\

    }



    if (parity) {

#define prev2 "prev"

#define next2 "cur"

        FILTER

#undef prev2

#undef next2

    } else {

#define prev2 "cur"

#define next2 "next"

        FILTER

#undef prev2

#undef next2

    }

}
