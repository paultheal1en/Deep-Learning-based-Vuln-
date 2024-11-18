uint64_t HELPER(abs_i64)(int64_t val)

{

    HELPER_LOG("%s: val 0x%" PRIx64 "\n", __func__, val);



    if (val < 0) {

        return -val;

    } else {

        return val;

    }

}
