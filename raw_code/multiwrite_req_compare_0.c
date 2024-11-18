static int multiwrite_req_compare(const void *a, const void *b)

{

    const BlockRequest *req1 = a, *req2 = b;



    /*

     * Note that we can't simply subtract req2->sector from req1->sector

     * here as that could overflow the return value.

     */

    if (req1->sector > req2->sector) {

        return 1;

    } else if (req1->sector < req2->sector) {

        return -1;

    } else {

        return 0;

    }

}
