static void show_chapters(WriterContext *w, AVFormatContext *fmt_ctx)

{

    int i;



    writer_print_section_header(w, SECTION_ID_CHAPTERS);

    for (i = 0; i < fmt_ctx->nb_chapters; i++) {

        AVChapter *chapter = fmt_ctx->chapters[i];



        writer_print_section_header(w, SECTION_ID_CHAPTER);

        print_int("id", chapter->id);

        print_q  ("time_base", chapter->time_base, '/');

        print_int("start", chapter->start);

        print_time("start_time", chapter->start, &chapter->time_base);

        print_int("end", chapter->end);

        print_time("end_time", chapter->end, &chapter->time_base);

        show_tags(w, chapter->metadata, SECTION_ID_CHAPTER_TAGS);

        writer_print_section_footer(w);

    }

    writer_print_section_footer(w);

}
