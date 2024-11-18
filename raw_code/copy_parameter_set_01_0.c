static void copy_parameter_set(void **to, void **from, int count, int size)

{

    int i;



    for (i = 0; i < count; i++) {

        if (to[i] && !from[i])

            av_freep(&to[i]);

        else if (from[i] && !to[i])

            to[i] = av_malloc(size);



        if (from[i])

            memcpy(to[i], from[i], size);

    }

}
