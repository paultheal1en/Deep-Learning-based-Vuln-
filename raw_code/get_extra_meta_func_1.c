static const ID3v2EMFunc *get_extra_meta_func(const char *tag, int isv34)

{

    int i = 0;

    while (ff_id3v2_extra_meta_funcs[i].tag3) {

        if (!memcmp(tag,

                    (isv34 ?

                        ff_id3v2_extra_meta_funcs[i].tag4 :

                        ff_id3v2_extra_meta_funcs[i].tag3),

                    (isv34 ? 4 : 3)))

            return &ff_id3v2_extra_meta_funcs[i];

        i++;

    }

    return NULL;

}
