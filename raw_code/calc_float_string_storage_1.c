static gsize calc_float_string_storage(double value)

{

    int whole_value = value;

    gsize i = 0;

    do {

        i++;

    } while (whole_value /= 10);

    return i + 2 + FLOAT_STRING_PRECISION;

}
