static inline int blah (int32_t i)

{

    if (i > 0x43c07fff)

        return 32767;

    else if (i < 0x43bf8000)

        return -32768;

    else

        return i - 0x43c00000;

}
