uint64_t HELPER(popcnt)(uint64_t r2)

{

    uint64_t ret = 0;

    int i;



    for (i = 0; i < 64; i += 8) {

        uint64_t t = ctpop32((r2 >> i) & 0xff);

        ret |= t << i;

    }

    return ret;

}
