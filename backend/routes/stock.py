from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import yfinance as yf
from datetime import datetime
from backend.models import StockData, User
from backend.schemas import StockDataResponse
from backend.database import get_db
from backend.utils.security import get_current_user  

from backend.models import User
from backend.database import get_db

router = APIRouter(tags=["Stock Data"])

@router.get("/stock/{symbol}", response_model=list[StockDataResponse])
def get_stock_data(
    symbol: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1mo")
        data = []
        
        for date, row in hist.iterrows():
            stock_data = StockData(
                symbol=symbol,
                date=date.date(),
                open_price=row['Open'],
                high_price=row['High'],
                low_price=row['Low'],
                close_price=row['Close'],
                volume=row['Volume']
            )
            db.add(stock_data)
            data.append({
                "symbol": symbol,
                "date": date.date().isoformat(),
                "open": row['Open'],
                "high": row['High'],
                "low": row['Low'],
                "close": row['Close'],
                "volume": row['Volume']
            })
        
        db.commit()
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))