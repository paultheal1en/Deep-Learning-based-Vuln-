static inline void code_gen_alloc(size_t tb_size)

{

    code_gen_buffer_size = size_code_gen_buffer(tb_size);

    code_gen_buffer = alloc_code_gen_buffer();

    if (code_gen_buffer == NULL) {

        fprintf(stderr, "Could not allocate dynamic translator buffer\n");

        exit(1);

    }



    map_exec(code_gen_prologue, sizeof(code_gen_prologue));

    code_gen_buffer_max_size = code_gen_buffer_size -

        (TCG_MAX_OP_SIZE * OPC_BUF_SIZE);

    code_gen_max_blocks = code_gen_buffer_size / CODE_GEN_AVG_BLOCK_SIZE;

    tbs = g_malloc(code_gen_max_blocks * sizeof(TranslationBlock));

}
