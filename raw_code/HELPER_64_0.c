uint32_t HELPER(servc)(uint32_t r1, uint64_t r2)

{

    if (sclp_service_call(env, r1, r2)) {

        return 3;

    }



    return 0;

}
