static void check_mct(uint8_t *ref0, uint8_t *ref1, uint8_t *ref2,

                      uint8_t *new0, uint8_t *new1, uint8_t *new2) {

    declare_func(void, void *src0, void *src1, void *src2, int csize);



    randomize_buffers();

    call_ref(ref0, ref1, ref2, BUF_SIZE / sizeof(int32_t));

    call_new(new0, new1, new2, BUF_SIZE / sizeof(int32_t));

    if (memcmp(ref0, new0, BUF_SIZE) || memcmp(ref1, new1, BUF_SIZE) ||

        memcmp(ref2, new2, BUF_SIZE))

        fail();

    bench_new(new0, new1, new2, BUF_SIZE / sizeof(int32_t));

}
