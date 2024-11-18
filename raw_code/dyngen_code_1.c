int dyngen_code(TCGContext *s, uint8_t *gen_code_buf)

{

#ifdef CONFIG_PROFILER

    {

        extern int64_t dyngen_op_count;

        extern int dyngen_op_count_max;

        int n;

        n = (gen_opc_ptr - gen_opc_buf);

        dyngen_op_count += n;

        if (n > dyngen_op_count_max)

            dyngen_op_count_max = n;

    }

#endif



    tcg_gen_code_common(s, gen_code_buf, 0, NULL);



    /* flush instruction cache */

    flush_icache_range((unsigned long)gen_code_buf, 

                       (unsigned long)s->code_ptr);

    return s->code_ptr -  gen_code_buf;

}
