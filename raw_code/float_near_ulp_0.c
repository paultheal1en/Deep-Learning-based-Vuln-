int float_near_ulp(float a, float b, unsigned max_ulp)

{

    union av_intfloat32 x, y;



    x.f = a;

    y.f = b;



    if (is_negative(x) != is_negative(y)) {

        // handle -0.0 == +0.0

        return a == b;

    }



    if (abs(x.i - y.i) <= max_ulp)

        return 1;



    return 0;

}
