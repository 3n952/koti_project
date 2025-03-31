from datetime import datetime, timedelta, timezone
import time


def dt2unix(date_str):
    # date_str형식 "2023-12-11 23:59:59"
    # datetime 객체로 변환
    # KST (UTC+9) 설정
    kst = timezone(timedelta(hours=9))
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=kst)

    # Unix 타임(초 단위)으로 변환
    unix_time = int(dt.timestamp())
    return unix_time

def ymd_spliter(ymd):
    '''
    ymd 20231211와 같은 형식 y, m, d로 분할
    '''

    if isinstance(ymd, str):
        y = ymd[:4]
        m = ymd[4:6]
        d = ymd[6:]
    
    else:
        ymd = ymd.astype(str)
        return ymd_spliter(ymd)
    
    return y, m, d

def moct_link_list_maker(moct_file_path):
    '''
    moct의 특정 행정구역에 속하는 링크 id만 뽑아서 txt파일에 저장
    '''
    pass

def recog_day(date_str: str, output_type: str = 'idx') -> datetime:
    '''
    str을 입력받아 해당 요일을 파악하는 코드
    '''
    # str을 datetime으로 변환 Convert string to datetime object
    date_obj = datetime.strptime(date_str, "%Y%m%d")

    # 요일 추출
    if output_type == 'idx':
        result = date_obj.weekday()
    else:
        result = date_obj.strftime("%A") 

    return result

def get_weekdays_in_same_week(date_str: str) -> list:
    """
    date string in 'YYYYMMDD' format이 주어지면 해당 날짜를 제외한 주중,주말에 대한 datetime 값을 반환함
    
    Args:
        date_str (str): The date string in 'YYYYMMDD' format.
    
    Returns:
        list: A list of up to 4 datetime objects (remaining weekdays in the same week).
    """
    date_obj = datetime.strptime(date_str, "%Y%m%d")  # Convert string to datetime
    weekday = date_obj.weekday()  # 0 = Monday, ..., 6 = Sunday

    # Get the Monday of the same week
    monday = date_obj - timedelta(days=weekday)

    # Generate all weekdays of the same week (Monday to Friday)
    weekdays = [monday + timedelta(days=i) for i in range(5)]  # Monday to Friday

    # Exclude the input date
    weekdays = [d for d in weekdays if d != date_obj]

    return weekdays  # Return the list of remaining weekdays

def corresponding_sample_timecode_unixtime(date, hours):
    """
    Convert date in 'YYYYMMDD' format to Unix timestamps for specific time.
    Args:
        date (str): str of date as integer (e.g., 20231211).
    
    Returns:
        integer: Unix timestamps corresponding to specific time of date.
    """
    date = str(date)
    dt = datetime.strptime(date, "%Y%m%d")  # YYYYMMDD -> datetime 객체
    # 시간 설정 integer 형태의 타임코드
    kst = timezone(timedelta(hours=9))
    dt = dt.replace(hour=hours, minute=0, second=0, tzinfo=kst)
    unix_time = int(dt.timestamp())

    return unix_time

if __name__ == '__main__':
    date = '20231211'
    hours = '07'
    hour = int(hours)
    print(hour)

    ut = corresponding_sample_timecode_unixtime(date, hour)
    print(ut)