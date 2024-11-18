static inline bool fp_access_check(DisasContext *s)

{

    assert(!s->fp_access_checked);

    s->fp_access_checked = true;



    if (s->cpacr_fpen) {

        return true;

    }



    gen_exception_insn(s, 4, EXCP_UDEF, syn_fp_access_trap(1, 0xe, false),

                       default_exception_el(s));

    return false;

}
