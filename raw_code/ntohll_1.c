static uint64_t ntohll(uint64_t v) {

    union { uint32_t lv[2]; uint64_t llv; } u;

    u.llv = v;

    return ((uint64_t)ntohl(u.lv[0]) << 32) | (uint64_t) ntohl(u.lv[1]);

}
