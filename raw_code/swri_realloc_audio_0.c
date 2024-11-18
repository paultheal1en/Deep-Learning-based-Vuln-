int swri_realloc_audio(AudioData *a, int count){

    int i, countb;

    AudioData old;



    if(count < 0 || count > INT_MAX/2/a->bps/a->ch_count)

        return AVERROR(EINVAL);



    if(a->count >= count)

        return 0;



    count*=2;



    countb= FFALIGN(count*a->bps, ALIGN);

    old= *a;



    av_assert0(a->bps);

    av_assert0(a->ch_count);



    a->data= av_mallocz_array(countb, a->ch_count);

    if(!a->data)

        return AVERROR(ENOMEM);

    for(i=0; i<a->ch_count; i++){

        a->ch[i]= a->data + i*(a->planar ? countb : a->bps);

        if(a->planar) memcpy(a->ch[i], old.ch[i], a->count*a->bps);

    }

    if(!a->planar) memcpy(a->ch[0], old.ch[0], a->count*a->ch_count*a->bps);

    av_freep(&old.data);

    a->count= count;



    return 1;

}
