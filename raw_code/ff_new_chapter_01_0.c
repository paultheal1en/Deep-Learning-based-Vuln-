AVChapter *ff_new_chapter(AVFormatContext *s, int id, AVRational time_base, int64_t start, int64_t end, const char *title)

{

    AVChapter *chapter = NULL;

    int i;



    for(i=0; i<s->nb_chapters; i++)

        if(s->chapters[i]->id == id)

            chapter = s->chapters[i];



    if(!chapter){

        chapter= av_mallocz(sizeof(AVChapter));

        if(!chapter)

            return NULL;

        dynarray_add(&s->chapters, &s->nb_chapters, chapter);

    }

    if(chapter->title)

        av_free(chapter->title);

    chapter->title = av_strdup(title);

    chapter->id    = id;

    chapter->time_base= time_base;

    chapter->start = start;

    chapter->end   = end;



    return chapter;

}
