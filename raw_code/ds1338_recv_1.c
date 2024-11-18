static int ds1338_recv(I2CSlave *i2c)

{

    DS1338State *s = FROM_I2C_SLAVE(DS1338State, i2c);

    uint8_t res;



    res  = s->nvram[s->ptr];

    s->ptr = (s->ptr + 1) & 0xff;

    return res;

}
