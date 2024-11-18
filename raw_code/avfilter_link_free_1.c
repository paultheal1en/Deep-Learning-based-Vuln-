void avfilter_link_free(AVFilterLink **link)

{

    if (!*link)

        return;



    if ((*link)->pool) {

        int i;

        for (i = 0; i < POOL_SIZE; i++) {

            if ((*link)->pool->pic[i]) {

                AVFilterBufferRef *picref = (*link)->pool->pic[i];

                /* free buffer: picrefs stored in the pool are not

                 * supposed to contain a free callback */

                av_freep(&picref->buf->data[0]);

                av_freep(&picref->buf);



                av_freep(&picref->audio);

                av_freep(&picref->video);

                av_freep(&picref);

            }

        }

        av_freep(&(*link)->pool);

    }

    av_freep(link);

}
