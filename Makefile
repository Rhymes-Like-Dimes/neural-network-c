# Makefile for neuralNetwork project

CC = gcc
CFLAGS = -g -Iinc
SRCS = src/main.c src/nn.c src/utils.c
OUT = build/main.exe

all: $(OUT)

$(OUT): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(OUT)

run: $(OUT)
	./$(OUT)

clean:
	del /q build\*.exe 2>nul || rm -f build/*.exe
