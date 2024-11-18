static size_t curl_size_cb(void *ptr, size_t size, size_t nmemb, void *opaque)

{

    CURLState *s = ((CURLState*)opaque);

    size_t realsize = size * nmemb;

    size_t fsize;



    if(sscanf(ptr, "Content-Length: %zd", &fsize) == 1) {

        s->s->len = fsize;

    }



    return realsize;

}
