static inline int check_fit_i32(uint32_t val, unsigned int bits)

{

    return ((val << (32 - bits)) >> (32 - bits)) == val;

}
