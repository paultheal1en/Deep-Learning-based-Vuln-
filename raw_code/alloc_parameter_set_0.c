alloc_parameter_set(H264Context *h, void **vec, const unsigned int id, const unsigned int max,

                    const size_t size, const char *name)

{

    if(id>=max) {

        av_log(h->s.avctx, AV_LOG_ERROR, "%s_id (%d) out of range\n", name, id);

        return NULL;

    }



    if(!vec[id]) {

        vec[id] = av_mallocz(size);

        if(vec[id] == NULL)

            av_log(h->s.avctx, AV_LOG_ERROR, "cannot allocate memory for %s\n", name);

    }

    return vec[id];

}
