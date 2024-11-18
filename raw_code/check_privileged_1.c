static inline void check_privileged(DisasContext *s)

{

    if (s->tb->flags & (PSW_MASK_PSTATE >> 32)) {

        gen_program_exception(s, PGM_PRIVILEGED);

    }

}
