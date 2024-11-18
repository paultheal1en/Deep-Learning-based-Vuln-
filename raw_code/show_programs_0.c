static void show_programs(WriterContext *w, AVFormatContext *fmt_ctx)

{

    int i;



    writer_print_section_header(w, SECTION_ID_PROGRAMS);

    for (i = 0; i < fmt_ctx->nb_programs; i++) {

        AVProgram *program = fmt_ctx->programs[i];

        if (!program)

            continue;

        show_program(w, fmt_ctx, program);

    }

    writer_print_section_footer(w);

}
