static uint32_t get_generic_seed(void)

{

    uint8_t tmp[120];

    struct AVSHA *sha = (void*)tmp;

    clock_t last_t  = 0;

    static uint64_t i = 0;

    static uint32_t buffer[512] = { 0 };

    unsigned char digest[20];

    uint64_t last_i = i;



    av_assert0(sizeof(tmp) >= av_sha_size);



    if(TEST){

        memset(buffer, 0, sizeof(buffer));

        last_i = i = 0;

    }else{

#ifdef AV_READ_TIME

        buffer[13] ^= AV_READ_TIME();

        buffer[41] ^= AV_READ_TIME()>>32;

#endif

    }



    for (;;) {

        clock_t t = clock();



        if (last_t == t) {

            buffer[i & 511]++;

        } else {

            buffer[++i & 511] += (t - last_t) % 3294638521U;

            if (last_i && i - last_i > 4 || i - last_i > 64 || TEST && i - last_i > 8)

                break;

        }

        last_t = t;

    }



    if(TEST)

        buffer[0] = buffer[1] = 0;



    av_sha_init(sha, 160);

    av_sha_update(sha, (const uint8_t *)buffer, sizeof(buffer));

    av_sha_final(sha, digest);

    return AV_RB32(digest) + AV_RB32(digest + 16);

}
