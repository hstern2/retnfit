CC = nvcc
CFLAGS = -g
OBJS = driver.o gn.o util.o array.o

driver: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

clean:
	rm -f *.o

# patterns
%.o: %.cu
	$(CC) $(CFLAGS) -o $@ -c $<
