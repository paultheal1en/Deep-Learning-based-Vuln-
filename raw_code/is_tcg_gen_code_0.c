bool is_tcg_gen_code(uintptr_t tc_ptr)

{

    /* This can be called during code generation, code_gen_buffer_max_size

       is used instead of code_gen_ptr for upper boundary checking */

    return (tc_ptr >= (uintptr_t)tcg_ctx.code_gen_buffer &&

            tc_ptr < (uintptr_t)(tcg_ctx.code_gen_buffer +

                    tcg_ctx.code_gen_buffer_max_size));

}
