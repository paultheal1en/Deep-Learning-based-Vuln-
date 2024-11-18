int dyngen_code_search_pc(TCGContext *s, uint8_t *gen_code_buf,

                          const uint8_t *searched_pc)

{

    return tcg_gen_code_common(s, gen_code_buf, 1, searched_pc);

}
