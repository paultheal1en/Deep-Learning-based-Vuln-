static inline float64 ucf64_itod(uint64_t i)

{

    union {

        uint64_t i;

        float64 d;

    } v;



    v.i = i;

    return v.d;

}
