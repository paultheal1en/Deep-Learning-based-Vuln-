uint32_t HELPER(clz)(uint32_t x)

{

    int count;

    for (count = 32; x; count--)

        x >>= 1;

    return count;

}
