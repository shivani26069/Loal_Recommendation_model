import sqlite3
import random
from datetime import datetime, timedelta

def create_customer_database():
    """Create a database with 200+ realistic customer records"""
    
    # Predefined PAN-like identifiers (not real PANs)
    pan_numbers = [
        "ABCDE1234F", "FGHIJ5678K", "KLMNO9012P", "QRSTU3456V", "WXYZL7890A",
        "BCDEF2345G", "HIJKL6789M", "NOPQR1234S", "STUVW5678X", "YZABC9012D",
        "CDEFG3456H", "IJKLM7890N", "OPQRS2345T", "UVWXY6789Z", "ABCDF0123E",
        "GHIJK4567L", "MNOPQ8901R", "STUVX5432Y", "YZABD6789C", "EFGHI0987J",
        "KLMNP3210Q", "QRSTU7654W", "VWXYZ1098B", "ABCEG2468F", "HIJKN5791M",
        "OPQRT9135S", "UVWXA6420Z", "BCDFH8642G", "IJKLO1357N", "PQRSU4680V",
        "WXYZB7913C", "CDEFJ0246H", "GHIKM5789L", "MNOPV2468R", "STUWZ6801Y",
        "ABCFI3579E", "DEFGK6802H", "HIJLN9024M", "OPQRU1357S", "TUVWX4680Z",
        "YZABG7913D", "CDEHI0246J", "FGKLM3579P", "NOPQS6802V", "RUVWZ9135B",
        "ABCGK1468F", "DEFJN4791I", "GHIMO7024L", "KLPQT0357R", "NSUVX3680Y",
        "QWZAB6913E", "CDFHJ9246H", "GIKLP2579M", "NORSV5802S", "TUXYZ8135Z",
        "ABDHK4691C", "EFIJN7924F", "HILOQ0257I", "MNPTW3580L", "RSVYZ6813O",
        "UWABF9146R", "CDGIK2479U", "FJMPR5702X", "ILOSV8035A", "NQTWY1368D",
        "SABCE4691G", "VFHIJ7924J", "YKLMO0257M", "BDGPQ3580P", "EHKSV6813S",
        "GKPTU9146V", "JMQRX2479Y", "LOSUW5702B", "ORVYZ8035E", "RTXAB1368H",
        "UWCDE4691K", "XAFGH7924N", "ZBIJK0257Q", "CDLMN3580T", "FGOPQ6813W",
        "HJRST9146Z", "KLUVW2479C", "MNXYZ5702F", "PQABC8035I", "SDEFG1368L",
        "VHIJK4691O", "YLMNO7924R", "BAPQR0257U", "DESTV3580X", "GHWXY6813A",
        "JKZAB9146D", "MNCDE2479G", "PQFGH5702J", "STIJK8035M", "VWLMN1368P",
        "YZOPQ4691S", "BCRST7924V", "EFUVW0257Y", "HIXYZ3580B", "KLABE6813E",
        "NOPCD9146H", "QREFG2479K", "TUHIJ5702N", "WXKLM8035Q", "ZANOP1368T",
        "CDQRS4691W", "FGTUV7924Z", "IJWXY0257C", "LMZAB3580F", "OPCDE6813I",
        "RSFGH9146L", "UVIJK2479O", "XYKLM5702R", "ABNOP8035U", "DEQRS1368X",
        "GHTUV4691A", "JKWXY7924D", "MNZAB0257G", "PQCDE3580J", "STEFGH813M",
        "VWIJK9146P", "YZKLM2479S", "BCNOP5702V", "EFQRS8035Y", "HITUV1368B",
        "KLWXY4691E", "NOZAB7924H", "QRCDE0257K", "TUFGH3580N", "WXIJK6813Q",
        "ZAKLM9146T", "CDNOP2479W", "FGQRS5702Z", "IJTUV8035C", "LMWXY1368F",
        "OPZAB4691I", "RSCDE7924L", "UVFGH0257O", "XYIJK3580R", "ABKLM6813U",
        "DENOP9146X", "GHQRS2479A", "JKTUV5702D", "MNWXY8035G", "PQZAB1368J",
        "STCDE4691M", "VWFGH7924P", "YZIJK0257S", "BCKLM3580V", "EFNOP6813Y",
        "HIQRS9146B", "KLTUV2479E", "NOWXY5702H", "QRZAB8035K", "TUCDE1368N",
        "WXFGH4691Q", "ZAIJK7924T", "CDKLM0257W", "FGNOP3580Z", "IJQRS6813C",
        "LMTUV9146F", "OPWXY2479I", "RSZAB5702L", "UVCDE8035O", "XYFGH1368R",
        "ABIJK4691U", "DEKLM7924X", "GHNOP0257A", "JKQRS3580D", "MNTUV6813G",
        "PQWXY9146J", "STZAB2479M", "VWCDE5702P", "YZFGH8035S", "BCIJK1368V",
        "EFKLM4691Y", "HINOP7924B", "KLQRS0257E", "NOTUV3580H", "QRWXY6813K",
        "TUZAB9146N", "WXCDE2479Q", "ZAFGH5702T", "CDIJK8035W", "FGKLM1368Z",
        "IJNOP4691C", "LMQRS7924F", "OPTUV0257I", "RSWXY3580L", "UVZAB6813O",
        "XYCDE9146R", "ABFGH2479U", "DEIJK5702X", "GHKLM8035A", "JKNOP1368D",
        "MNQRS4691G", "PQTUV7924J", "STWXY0257M", "VWZAB3580P", "YZCDE6813S"
    ]
    
    # Indian names dataset
    indian_names = [
        "Rajesh Kumar", "Priya Sharma", "Amit Singh", "Sunita Patel", "Vikram Rao",
        "Kavita Mehta", "Ravi Gupta", "Meera Joshi", "Suresh Reddy", "Deepika Iyer",
        "Anil Verma", "Pooja Agarwal", "Sanjay Mishra", "Rekha Tiwari", "Manoj Yadav",
        "Neha Bansal", "Rohit Saxena", "Anita Chauhan", "Ajay Pandey", "Shweta Jain",
        "Rahul Srivastava", "Kiran Nair", "Sachin Malhotra", "Geeta Bhatia", "Vinod Khanna",
        "Seema Kapoor", "Deepak Arora", "Nisha Sethi", "Ramesh Chopra", "Preeti Bhardwaj",
        "Mahesh Gupta", "Ritu Singhal", "Ashok Agrawal", "Savita Dubey", "Dinesh Sinha",
        "Usha Thakur", "Pramod Shukla", "Meenakshi Rana", "Rakesh Jha", "Sunita Goyal",
        "Sandeep Rastogi", "Pooja Mittal", "Arun Kashyap", "Neelam Choudhary", "Sunil Tripathi",
        "Rita Dixit", "Mukesh Pathak", "Anjali Gupta", "Yogesh Awasthi", "Kamala Devi",
        "Naresh Pandey", "Sudha Sharma", "Girish Agarwal", "Mamta Singh", "Rajesh Tiwari",
        "Shobha Verma", "Devendra Mishra", "Sushma Yadav", "Prakash Bansal", "Renu Saxena",
        "Satish Chauhan", "Veena Pandey", "Mohan Jain", "Lata Srivastava", "Bharat Nair",
        "Pushpa Malhotra", "Kishore Bhatia", "Sarita Khanna", "Umesh Kapoor", "Nirmala Arora",
        "Jagdish Sethi", "Vandana Chopra", "Subhash Bhardwaj", "Purnima Gupta", "Ramakant Singhal",
        "Shanti Agrawal", "Govind Dubey", "Madhuri Sinha", "Brijesh Thakur", "Kamala Shukla",
        "Narayan Rana", "Urmila Jha", "Shyam Goyal", "Sita Rastogi", "Hari Mittal",
        "Radha Kashyap", "Krishnan Choudhary", "Gita Tripathi", "Murali Dixit", "Lakshmi Pathak",
        "Venkatesh Gupta", "Saroj Awasthi", "Balaji Devi", "Parvati Pandey", "Ganesh Sharma",
        "Saraswati Agarwal", "Mahesh Singh", "Durga Tiwari", "Vishnu Verma", "Shakti Mishra",
        "Indra Yadav", "Varun Bansal", "Gayatri Saxena", "Arjun Chauhan", "Sita Pandey",
        "Bhim Jain", "Kali Srivastava", "Ram Nair", "Shiva Malhotra", "Hanuman Bhatia",
        "Jagat Khanna", "Surya Kapoor", "Chandra Arora", "Agni Sethi", "Vayu Chopra",
        "Prithvi Bhardwaj", "Akash Gupta", "Jal Singhal", "Tejas Agrawal", "Veer Dubey",
        "Yash Sinha", "Dev Thakur", "Tej Shukla", "Arya Rana", "Ved Jha",
        "Som Goyal", "Ish Rastogi", "Harsh Mittal", "Kush Kashyap", "Luv Choudhary",
        "Neel Tripathi", "Om Dixit", "Prem Pathak", "Roop Gupta", "Shaan Awasthi",
        "Tanmay Devi", "Uday Pandey", "Vikas Sharma", "Yogi Agarwal", "Arpit Singh",
        "Bhuvan Tiwari", "Chirag Verma", "Dhanush Mishra", "Eshaan Yadav", "Faiz Bansal",
        "Gaurav Saxena", "Hitesh Chauhan", "Ishaan Pandey", "Jai Jain", "Karan Srivastava",
        "Lakhan Nair", "Manan Malhotra", "Naman Bhatia", "Ojas Khanna", "Parth Kapoor",
        "Qasim Arora", "Rihan Sethi", "Shivansh Chopra", "Tanvi Bhardwaj", "Utkarsh Gupta",
        "Vivaan Singhal", "Wyatt Agrawal", "Xander Dubey", "Yuvraj Sinha", "Zain Thakur",
        "Aarav Shukla", "Bhavya Rana", "Chirag Jha", "Dhruv Goyal", "Eeshan Rastogi",
        "Farid Mittal", "Gagan Kashyap", "Harsh Choudhary", "Ishan Tripathi", "Jatin Dixit",
        "Kartik Pathak", "Lavanya Gupta", "Manav Awasthi", "Nikhil Devi", "Omprakash Pandey",
        "Prashant Sharma", "Quincy Agarwal", "Rohan Singh", "Siddharth Tiwari", "Tanush Verma",
        "Urvashi Mishra", "Vaibhav Yadav", "Waman Bansal", "Xavi Saxena", "Yash Chauhan",
        "Zara Pandey", "Arsh Jain", "Bodhi Srivastava", "Cyrus Nair", "Diya Malhotra",
        "Evaan Bhatia", "Faisal Khanna", "Garima Kapoor", "Hrithik Arora", "Ishika Sethi",
        "Jiya Chopra", "Kabir Bhardwaj", "Lila Gupta", "Maya Singhal", "Naina Agrawal",
        "Odin Dubey", "Pari Sinha", "Qira Thakur", "Riya Shukla", "Sara Rana",
        "Tara Jha", "Uma Goyal", "Vanya Rastogi", "Wanda Mittal", "Xara Kashyap",
        "Yara Choudhary", "Zoya Tripathi", "Aisha Dixit", "Bhumi Pathak", "Chhavi Gupta"
    ]
    
    # Create database connection
    conn = sqlite3.connect('user_database.db')
    cursor = conn.cursor()
    
    # Drop existing table and create new one
    cursor.execute('DROP TABLE IF EXISTS users')
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pan_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            annual_income INTEGER NOT NULL,
            cibil_score INTEGER NOT NULL,
            previous_loan_status TEXT NOT NULL,
            created_date TEXT NOT NULL
        )
    ''')
    
    # Generate 200 customer records
    customers = []
    loan_statuses = ["Cleared", "Not Applicable", "Ongoing", "Defaulted"]
    
    for i in range(200):
        pan = pan_numbers[i]
        name = indian_names[i]
        age = random.randint(22, 65)
        
        # Generate income based on age (older people tend to have higher income)
        if age < 30:
            income = random.randint(300000, 800000)
        elif age < 40:
            income = random.randint(500000, 1200000)
        elif age < 50:
            income = random.randint(700000, 1800000)
        else:
            income = random.randint(600000, 2000000)
        
        # Generate CIBIL score (weighted towards good scores)
        cibil_weights = [0.05, 0.15, 0.25, 0.35, 0.20]  # Very Poor, Poor, Fair, Good, Excellent
        cibil_ranges = [(300, 549), (550, 649), (650, 699), (700, 749), (750, 850)]
        cibil_range = random.choices(cibil_ranges, weights=cibil_weights)[0]
        cibil = random.randint(cibil_range[0], cibil_range[1])
        
        # Generate previous loan status (weighted towards good borrowers)
        status_weights = [0.45, 0.30, 0.15, 0.10]  # Cleared, Not Applicable, Ongoing, Defaulted
        prev_loan = random.choices(loan_statuses, weights=status_weights)[0]
        
        # Create date (random date in last 2 years)
        start_date = datetime.now() - timedelta(days=730)
        random_days = random.randint(0, 730)
        created_date = (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d %H:%M:%S')
        
        customers.append((pan, name, age, income, cibil, prev_loan, created_date))
    
    # Insert all customers
    cursor.executemany('''
        INSERT INTO users (pan_number, name, age, annual_income, cibil_score, previous_loan_status, created_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', customers)
    
    conn.commit()
    
    # Display statistics
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    
    cursor.execute('SELECT previous_loan_status, COUNT(*) FROM users GROUP BY previous_loan_status')
    status_counts = cursor.fetchall()
    
    cursor.execute('SELECT AVG(cibil_score), MIN(cibil_score), MAX(cibil_score) FROM users')
    cibil_stats = cursor.fetchone()
    
    cursor.execute('SELECT AVG(annual_income), MIN(annual_income), MAX(annual_income) FROM users')
    income_stats = cursor.fetchone()
    
    conn.close()
    
    print(f"✅ Successfully created database with {total_users} customers!")
    print("\n📊 Database Statistics:")
    print(f"CIBIL Scores - Average: {cibil_stats[0]:.0f}, Min: {cibil_stats[1]}, Max: {cibil_stats[2]}")
    print(f"Annual Income - Average: ₹{income_stats[0]:,.0f}, Min: ₹{income_stats[1]:,.0f}, Max: ₹{income_stats[2]:,.0f}")
    print("\n🏦 Loan Status Distribution:")
    for status, count in status_counts:
        print(f"  {status}: {count} customers ({count/total_users*100:.1f}%)")

if __name__ == "__main__":
    create_customer_database()