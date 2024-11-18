static void *ff_realloc_static(void *ptr, unsigned int size)

{

    int i;

    if(!ptr)

      return av_mallocz_static(size);

    /* Look for the old ptr */

    for(i = 0; i < last_static; i++) {

        if(array_static[i] == ptr) {

            array_static[i] = av_realloc(array_static[i], size);

            return array_static[i];

        }

    }

    return NULL;



}
