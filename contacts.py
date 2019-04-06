class Contacts
    def __init__(self, name, phone, email, addr):
        self.name = name
        self.phone = phone
        self.email = email
        self.addr = addr

    def print_info(self):
        print("이름:", self.name)
        print("전화번호:", self.phone)
        print("이메일:", self.email)
        print("주소:", self.addr)

    @staticmethod
    def set_contact(self):
        name = input("이름:")
        phone = input("전화번호:")
        email = input("이메일:")
        addr = input("주소:")
        contact = Contacts(name, phone, email, addr)

    @staticmethod
    def get_contact(self, list, name):
        for i in list:
            i.print_info()

    @staticmethod
    def del_contact(list, name):
        for i, t in enumerate(list):
            if t.name == name:
                del list[i]

    @staticmethod
    def print_menu():
        print("1. 연락처 입력")
        print("2. 연락처 출력")
        print("3. 종료")
        menu = input("메뉴선택:")
        return int(menu)

    @staticmethod
    def run():
        list = []
        while 1:
            menu = Contacts.print_menu()
            if menu == 1:
                t = Contacts.set_contact()
                list.append()
            elif menu == 2:
                Contacts.get_contacts(list)
            elif menu == 3:
                name = input("삭제할 이름")
                Contacts.del_contact(list, name)
            elif menu == 4:
                break



