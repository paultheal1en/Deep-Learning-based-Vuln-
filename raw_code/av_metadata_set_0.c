int av_metadata_set(AVMetadata **pm, const char *key, const char *value)

{

    AVMetadata *m= *pm;

    AVMetadataTag *tag= av_metadata_get(m, key, NULL, AV_METADATA_MATCH_CASE);



    if(!m)

        m=*pm= av_mallocz(sizeof(*m));



    if(tag){

        av_free(tag->value);

        av_free(tag->key);

        *tag= m->elems[--m->count];

    }else{

        AVMetadataTag *tmp= av_realloc(m->elems, (m->count+1) * sizeof(*m->elems));

        if(tmp){

            m->elems= tmp;

        }else

            return AVERROR(ENOMEM);

    }

    if(value){

        m->elems[m->count].key  = av_strdup(key  );

        m->elems[m->count].value= av_strdup(value);

        m->count++;

    }

    if(!m->count) {

        av_free(m->elems);

        av_freep(pm);

    }



    return 0;

}
