int32_t HELPER(sdiv)(int32_t num, int32_t den)

{

    if (den == 0)

      return 0;



    return num / den;

}