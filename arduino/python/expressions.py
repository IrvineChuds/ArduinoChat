import bridge

@bridge.call()
def set_state(c: str) -> None:
    pass

def main():
    c = input()
    while c:
        set_state(c)
        c = input()

if __name__ == "__main__":
    main()