static int decode_hrd(VC9Context *v, GetBitContext *gb)

{

    int i, num;



    num = get_bits(gb, 5);



    if (v->hrd_rate || num != v->hrd_num_leaky_buckets)

    {

        av_freep(&v->hrd_rate);

    }

    if (!v->hrd_rate) v->hrd_rate = av_malloc(num);

    if (!v->hrd_rate) return -1;



    if (v->hrd_buffer || num != v->hrd_num_leaky_buckets)

    {

        av_freep(&v->hrd_buffer);

    }

    if (!v->hrd_buffer) v->hrd_buffer = av_malloc(num);

    if (!v->hrd_buffer) return -1;



    v->hrd_num_leaky_buckets = num;



    //exponent in base-2 for rate

    v->bit_rate_exponent = get_bits(gb, 4);

    //exponent in base-2 for buffer_size

    v->buffer_size_exponent = get_bits(gb, 4);



    for (i=0; i<num; i++)

    {

        //mantissae, ordered (if not, use a function ?

        v->hrd_rate[i] = get_bits(gb, 16);

        if (i && v->hrd_rate[i-1]>=v->hrd_rate[i])

        {

            av_log(v, AV_LOG_ERROR, "HDR Rates aren't strictly increasing:"

                   "%i vs %i\n", v->hrd_rate[i-1], v->hrd_rate[i]);

            return -1;

        }

        v->hrd_buffer[i] = get_bits(gb, 16);

        if (i && v->hrd_buffer[i-1]<v->hrd_buffer[i])

        {

            av_log(v, AV_LOG_ERROR, "HDR Buffers aren't decreasing:"

                   "%i vs %i\n", v->hrd_buffer[i-1], v->hrd_buffer[i]);

            return -1;

        }

    }

    return 0;

}
