uint32_t HELPER(rrbe)(uint32_t r1, uint64_t r2)

{

    if (r2 > ram_size) {

        return 0;

    }



    /* XXX implement */

#if 0

    env->storage_keys[r2 / TARGET_PAGE_SIZE] &= ~SK_REFERENCED;

#endif



    /*

     * cc

     *

     * 0  Reference bit zero; change bit zero

     * 1  Reference bit zero; change bit one

     * 2  Reference bit one; change bit zero

     * 3  Reference bit one; change bit one

     */

    return 0;

}
