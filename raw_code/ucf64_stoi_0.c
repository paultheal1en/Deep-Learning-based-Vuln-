static inline uint32_t ucf64_stoi(float32 s)

{

    union {

        uint32_t i;

        float32 s;

    } v;



    v.s = s;

    return v.i;

}
