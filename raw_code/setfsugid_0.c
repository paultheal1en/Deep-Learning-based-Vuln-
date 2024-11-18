static int setfsugid(int uid, int gid)

{

    /*

     * We still need DAC_OVERRIDE because  we don't change

     * supplementary group ids, and hence may be subjected DAC rules

     */

    cap_value_t cap_list[] = {

        CAP_DAC_OVERRIDE,

    };



    setfsgid(gid);

    setfsuid(uid);



    if (uid != 0 || gid != 0) {

        return do_cap_set(cap_list, ARRAY_SIZE(cap_list), 0);

    }

    return 0;

}
