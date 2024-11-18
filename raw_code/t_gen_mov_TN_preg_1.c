static inline void t_gen_mov_TN_preg(TCGv tn, int r)

{

    if (r < 0 || r > 15) {

        fprintf(stderr, "wrong register read $p%d\n", r);

    }

    if (r == PR_BZ || r == PR_WZ || r == PR_DZ) {

        tcg_gen_mov_tl(tn, tcg_const_tl(0));

    } else if (r == PR_VR) {

        tcg_gen_mov_tl(tn, tcg_const_tl(32));

    } else {

        tcg_gen_mov_tl(tn, cpu_PR[r]);

    }

}
