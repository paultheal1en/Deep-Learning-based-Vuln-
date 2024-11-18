int qemu_timeout_ns_to_ms(int64_t ns)

{

    int64_t ms;

    if (ns < 0) {

        return -1;

    }



    if (!ns) {

        return 0;

    }



    /* Always round up, because it's better to wait too long than to wait too

     * little and effectively busy-wait

     */

    ms = DIV_ROUND_UP(ns, SCALE_MS);



    /* To avoid overflow problems, limit this to 2^31, i.e. approx 25 days */

    if (ms > (int64_t) INT32_MAX) {

        ms = INT32_MAX;

    }



    return (int) ms;

}
