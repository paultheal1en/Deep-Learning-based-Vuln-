void tcg_dump_info(FILE *f,

                   int (*cpu_fprintf)(FILE *f, const char *fmt, ...))

{

    TCGContext *s = &tcg_ctx;

    int64_t tot;



    tot = s->interm_time + s->code_time;

    cpu_fprintf(f, "JIT cycles          %" PRId64 " (%0.3f s at 2.4 GHz)\n",

                tot, tot / 2.4e9);

    cpu_fprintf(f, "translated TBs      %" PRId64 " (aborted=%" PRId64 " %0.1f%%)\n", 

                s->tb_count, 

                s->tb_count1 - s->tb_count,

                s->tb_count1 ? (double)(s->tb_count1 - s->tb_count) / s->tb_count1 * 100.0 : 0);

    cpu_fprintf(f, "avg ops/TB          %0.1f max=%d\n", 

                s->tb_count ? (double)s->op_count / s->tb_count : 0, s->op_count_max);

    cpu_fprintf(f, "deleted ops/TB      %0.2f\n",

                s->tb_count ? 

                (double)s->del_op_count / s->tb_count : 0);

    cpu_fprintf(f, "avg temps/TB        %0.2f max=%d\n",

                s->tb_count ? 

                (double)s->temp_count / s->tb_count : 0,

                s->temp_count_max);

    

    cpu_fprintf(f, "cycles/op           %0.1f\n", 

                s->op_count ? (double)tot / s->op_count : 0);

    cpu_fprintf(f, "cycles/in byte      %0.1f\n", 

                s->code_in_len ? (double)tot / s->code_in_len : 0);

    cpu_fprintf(f, "cycles/out byte     %0.1f\n", 

                s->code_out_len ? (double)tot / s->code_out_len : 0);

    if (tot == 0)

        tot = 1;

    cpu_fprintf(f, "  gen_interm time   %0.1f%%\n", 

                (double)s->interm_time / tot * 100.0);

    cpu_fprintf(f, "  gen_code time     %0.1f%%\n", 

                (double)s->code_time / tot * 100.0);

    cpu_fprintf(f, "liveness/code time  %0.1f%%\n", 

                (double)s->la_time / (s->code_time ? s->code_time : 1) * 100.0);

    cpu_fprintf(f, "cpu_restore count   %" PRId64 "\n",

                s->restore_count);

    cpu_fprintf(f, "  avg cycles        %0.1f\n",

                s->restore_count ? (double)s->restore_time / s->restore_count : 0);



    dump_op_count();

}
